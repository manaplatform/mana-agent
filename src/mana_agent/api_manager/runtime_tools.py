"""Narrow structured API tools exposed to the chat/gateway runtime."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mana_agent.api_manager.documentation import SemanticDefinition
from mana_agent.api_manager.discovery import ApiRouteDecision
from mana_agent.api_manager.models import AuthenticationConfig
from mana_agent.api_manager.service import ApiManagerService, safe_result
from mana_agent.api_manager.events import api_event_scope


_SERVICES: dict[str, ApiManagerService] = {}
_SERVICES_LOCK = threading.RLock()


class _Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_decision_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


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


class _Import(_Decision):
    name: str = Field(min_length=1, max_length=160)
    text: str = Field(default="", max_length=10 * 1024 * 1024)
    path: str = ""
    url: str = ""
    semantic_definition: SemanticDefinition | None = None
    save: bool = True
    ephemeral: bool = False
    refresh_integration_id: str = Field(
        default="", pattern=r"^(|api_[a-f0-9]{24})$"
    )

    @model_validator(mode="after")
    def exactly_one_source(self) -> "_Import":
        if sum(bool(item) for item in (self.text, self.path, self.url)) != 1:
            raise ValueError("Select exactly one of text, path, or url.")
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
) -> list[Any]:
    resolved_root = str(Path(root).expanduser().resolve())
    manager = service or _service_for_root(resolved_root)

    def encode(operation: Any, *, session_id: str, source_decision_id: str) -> str:
        return _json(
            operation,
            root=resolved_root,
            session_id=session_id,
            source_decision_id=source_decision_id,
        )

    def import_docs(**values: Any) -> str:
        request = _Import(**values)
        return _json(
            lambda: manager.import_documentation(
                name=request.name,
                source_decision_id=request.source_decision_id,
                text=request.text,
                path=request.path,
                url=request.url,
                semantic_definition=request.semantic_definition,
                save=request.save,
                ephemeral=request.ephemeral,
                refresh_integration_id=request.refresh_integration_id,
            ),
            root=resolved_root,
            session_id=request.session_id,
            source_decision_id=request.source_decision_id,
        )

    def import_semantic_docs(**values: Any) -> str:
        request = _SemanticImport(**values)
        return _json(
            lambda: manager.import_documentation(
                name=request.name,
                source_decision_id=request.source_decision_id,
                text=request.text,
                text_reference=request.documentation_reference,
                semantic_definition=request.semantic_definition,
                save=request.save,
                ephemeral=request.ephemeral,
                refresh_integration_id=request.refresh_integration_id,
            ),
            root=resolved_root,
            session_id=request.session_id,
            source_decision_id=request.source_decision_id,
        )

    def preview(**values: Any) -> str:
        request = _Request(**values)
        if request.routing_decision.source_decision_id != request.source_decision_id:
            raise ValueError("Routing decision ID does not match the tool decision ID.")
        return _json(
            lambda: manager.preview_request(
                session_id=request.session_id,
                source_decision_id=request.source_decision_id,
                routing_decision=request.routing_decision,
                **request.request_kwargs(),
            ),
            root=resolved_root,
            session_id=request.session_id,
            source_decision_id=request.source_decision_id,
        )

    def execute(**values: Any) -> str:
        request = _Execute(**values)
        if request.routing_decision.source_decision_id != request.source_decision_id:
            raise ValueError("Routing decision ID does not match the tool decision ID.")
        return _json(
            lambda: manager.execute_request(
                approval_reference=request.approval_reference,
                session_id=request.session_id,
                routing_decision=request.routing_decision,
                **request.request_kwargs(),
            ),
            root=resolved_root,
            session_id=request.session_id,
            source_decision_id=request.source_decision_id,
        )

    return [
        StructuredTool.from_function(
            name="api_workflow_decide",
            description=(
                "Record the model's strict ordered API workflow decision. This must be the first "
                "API-route tool call; completion is validated against its required_actions."
            ),
            args_schema=_WorkflowDecision,
            func=lambda **values: encode(
                lambda: _WorkflowDecision(**values).model_dump(mode="json"),
                session_id=str(values["session_id"]),
                source_decision_id=str(values["source_decision_id"]),
            ),
        ),
        StructuredTool.from_function(
            name="api_docs_inspect",
            description=(
                "Read one authorized API documentation URL, workspace file, or pasted text source "
                "through the API network/file policy. Returns source evidence only and never infers, "
                "imports, or executes an operation."
            ),
            args_schema=_Inspect,
            func=lambda source_decision_id, session_id, text="", path="", url="": encode(
                lambda: manager.inspect_documentation(text=text, path=path, url=url),
                session_id=session_id,
                source_decision_id=source_decision_id,
            ),
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
            func=lambda source_decision_id, session_id, include_disabled=True: encode(
                lambda: manager.list_integrations(include_disabled=include_disabled),
                session_id=session_id,
                source_decision_id=source_decision_id,
            ),
        ),
        StructuredTool.from_function(
            name="api_integration_get",
            description="Inspect one saved normalized API integration without credential values.",
            args_schema=_IntegrationId,
            func=lambda integration_id, source_decision_id, session_id: encode(
                lambda: manager.get_integration(integration_id),
                session_id=session_id,
                source_decision_id=source_decision_id,
            ),
        ),
        StructuredTool.from_function(
            name="api_integration_update",
            description="Update only explicit mutable integration metadata fields.",
            args_schema=_Update,
            func=lambda integration_id, source_decision_id, session_id, name=None, description=None, enabled=None, authentication=None: encode(
                lambda: manager.update_integration(
                    integration_id,
                    name=name,
                    description=description,
                    enabled=enabled,
                    authentication=authentication,
                ),
                session_id=session_id,
                source_decision_id=source_decision_id,
            ),
            metadata={"transactional_adapter": "api_integration"},
        ),
        StructuredTool.from_function(
            name="api_integration_delete",
            description="Delete one saved integration only with explicit delete intent.",
            args_schema=_Delete,
            func=lambda integration_id, explicit, source_decision_id, session_id: encode(
                lambda: manager.registry.delete(integration_id, explicit=explicit),
                session_id=session_id,
                source_decision_id=source_decision_id,
            ),
            metadata={"transactional_adapter": "api_integration"},
        ),
        StructuredTool.from_function(
            name="api_operations_search",
            description=(
                "Retrieve and rank operations from enabled integrations. This returns candidates only; "
                "a structured model decision must select an operation before preview or execution."
            ),
            args_schema=_Search,
            func=lambda query, limit, source_decision_id, session_id: encode(
                lambda: [
                    item.model_dump(mode="json")
                    for item in manager.discovery.search(query, limit=limit)
                ],
                session_id=session_id,
                source_decision_id=source_decision_id,
            ),
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
