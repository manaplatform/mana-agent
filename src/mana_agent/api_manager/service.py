"""High-level API Manager orchestration used by gateway tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mana_agent.api_manager.discovery import ApiOperationDiscovery, ApiRouteDecision
from mana_agent.api_manager.documentation import DocumentationImporter, SemanticDefinition
from mana_agent.api_manager.errors import ApiManagerError, UpstreamApiError
from mana_agent.api_manager.executor import (
    ApiExecutor,
    NetworkAccessPolicy,
    PendingApiApprovalBroker,
    validate_network_target,
)
from mana_agent.api_manager.registry import ApiIntegrationRegistry
from mana_agent.api_manager.request_builder import ApiRequestBuilder
from mana_agent.api_manager.models import AuthenticationConfig
from mana_agent.api_manager.events import publish_api_event
from mana_agent.config.settings import Settings, mana_home


class ApiManagerService:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        registry: ApiIntegrationRegistry | None = None,
        executor: ApiExecutor | None = None,
        network_policy: NetworkAccessPolicy | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.registry = registry or ApiIntegrationRegistry(event_sink=publish_api_event)
        self.approvals = (
            executor.approval_broker
            if executor is not None
            and isinstance(executor.approval_broker, PendingApiApprovalBroker)
            else PendingApiApprovalBroker()
        )
        settings = Settings()
        resolved_network_policy = network_policy or NetworkAccessPolicy(
            allowed_hosts=_csv(getattr(settings, "mana_api_manager_allowed_hosts", "")),
            trusted_internal_hosts=_csv(
                getattr(settings, "mana_api_manager_trusted_internal_hosts", "")
            ),
            trusted_internal_networks=_csv(
                getattr(settings, "mana_api_manager_trusted_internal_networks", "")
            ),
            allow_http=bool(getattr(settings, "mana_api_manager_allow_http", False)),
            maximum_redirects=int(
                getattr(settings, "mana_api_manager_max_redirects", 3)
            ),
            maximum_response_bytes=int(
                getattr(settings, "mana_api_manager_max_response_bytes", 10_485_760)
            ),
        )
        self.executor = executor or ApiExecutor(
            network_policy=resolved_network_policy,
            approval_broker=self.approvals,
            event_sink=publish_api_event,
            artifact_directory=mana_home() / "api_manager" / "artifacts",
        )
        self.importer = DocumentationImporter(
            allowed_file_roots=(self.workspace_root,),
            fetcher=self.executor,
        )
        self.builder = ApiRequestBuilder(self.registry)
        self.discovery = ApiOperationDiscovery(self.registry)

    def import_documentation(
        self,
        *,
        name: str,
        source_decision_id: str,
        text: str = "",
        path: str = "",
        url: str = "",
        semantic_definition: SemanticDefinition | dict[str, Any] | None = None,
        save: bool = True,
        ephemeral: bool = False,
        refresh_integration_id: str = "",
    ) -> dict[str, Any]:
        publish_api_event(
            "api.documentation.import.started",
            {"name": name, "source_kind": "file" if path else "url" if url else "text"},
        )
        selected_sources = sum(bool(item) for item in (text, path, url))
        if selected_sources != 1:
            raise ValueError("Select exactly one documentation source: text, path, or URL.")
        try:
            if path:
                integration = self.importer.from_file(
                    path,
                    name=name,
                    source_decision_id=source_decision_id,
                    semantic_definition=semantic_definition,
                )
            elif url:
                integration = self.importer.from_url(
                    url,
                    name=name,
                    source_decision_id=source_decision_id,
                    semantic_definition=semantic_definition,
                )
            else:
                integration = self.importer.from_text(
                    text,
                    name=name,
                    source_decision_id=source_decision_id,
                    semantic_definition=semantic_definition,
                )
        except (ApiManagerError, ValueError, OSError, UnicodeError) as exc:
            publish_api_event(
                "api.documentation.import.failed",
                {"name": name, "error_code": getattr(exc, "code", "invalid_documentation")},
            )
            raise
        integration = integration.model_copy(update={"ephemeral": ephemeral})
        if ephemeral:
            integration = self.registry.save_ephemeral(integration)
        elif save:
            integration = (
                self.registry.refresh(refresh_integration_id, integration)
                if refresh_integration_id
                else self.registry.save(integration)
            )
        result = {
            "saved": bool(save and not ephemeral),
            "operation_count": len(integration.operations),
            "unresolved_fields": sorted(
                {
                    field
                    for operation in integration.operations
                    for field in operation.unresolved_fields
                }
            ),
            "integration": integration.model_dump(mode="json", by_alias=True),
        }
        publish_api_event(
            "api.documentation.import.completed",
            {
                "integration_id": integration.integration_id,
                "operation_count": len(integration.operations),
                "saved": result["saved"],
            },
        )
        return result

    def inspect_documentation(
        self,
        *,
        text: str = "",
        path: str = "",
        url: str = "",
    ) -> dict[str, Any]:
        """Read one authorized documentation source without inferring API semantics."""
        if sum(bool(item) for item in (text, path, url)) != 1:
            raise ValueError("Select exactly one documentation source: text, path, or URL.")
        reference = "pasted-text"
        content_type = "text/plain"
        if path:
            resolved = Path(path).expanduser().resolve()
            if (
                resolved != self.workspace_root
                and self.workspace_root not in resolved.parents
            ):
                raise ValueError("Documentation file is outside the authorized workspace root.")
            payload = resolved.read_bytes()
            reference = str(resolved)
        elif url:
            payload, content_type = self.executor.fetch_documentation(url)
            reference = url
        else:
            payload = text.encode("utf-8")
        if len(payload) > 10 * 1024 * 1024:
            raise ValueError("Documentation exceeds the 10 MiB inspection limit.")
        return {
            "reference": reference,
            "content_type": content_type,
            "text": payload.decode("utf-8", errors="strict"),
            "bytes": len(payload),
            "truncated": False,
        }

    def list_integrations(self, *, include_disabled: bool = True) -> list[dict[str, Any]]:
        return [
            {
                "integration_id": item.integration_id,
                "name": item.name,
                "description": item.description,
                "enabled": item.enabled,
                "active_version": item.active_version,
                "operations": len(item.operations),
                "servers": [server.model_dump(mode="json") for server in item.servers],
            }
            for item in self.registry.list(include_disabled=include_disabled)
        ]

    def get_integration(self, integration_id: str) -> dict[str, Any]:
        return self.registry.get(integration_id).model_dump(mode="json", by_alias=True)

    def update_integration(
        self,
        integration_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
        authentication: list[AuthenticationConfig] | None = None,
    ) -> dict[str, Any]:
        changes = {
            key: value
            for key, value in {
                "name": name,
                "description": description,
                "enabled": enabled,
                "authentication": tuple(authentication) if authentication is not None else None,
            }.items()
            if value is not None
        }
        if not changes:
            raise ValueError("At least one explicit integration change is required.")
        return self.registry.update(integration_id, changes).model_dump(mode="json", by_alias=True)

    def preview_request(
        self,
        *,
        routing_decision: ApiRouteDecision | dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        route = ApiRouteDecision.model_validate(routing_decision)
        if route.workflow not in {"request_preview", "request_execution"}:
            raise ValueError("The model decision did not select a request preview workflow.")
        evidence = self._validate_route(route)
        if (
            evidence.integration_id != kwargs.get("integration_id")
            or evidence.operation_id != kwargs.get("operation_id")
        ):
            raise ValueError(
                "Request identifiers do not match the validated model operation decision."
            )
        request = self._build_validated(**kwargs)
        preview = self.builder.preview(request)
        publish_api_event(
            "api.operation.selected",
            {
                "integration_id": request.integration_id,
                "operation_id": request.operation_id,
                "method": request.method,
                "risk_level": request.risk_level.value,
                "routing_evidence": evidence.model_dump(mode="json"),
            },
        )
        publish_api_event(
            "api.request.validation.completed",
            {
                "integration_id": request.integration_id,
                "operation_id": request.operation_id,
                "valid": True,
            },
        )
        return preview.model_dump(mode="json")

    def execute_request(
        self,
        *,
        approval_reference: str = "",
        session_id: str = "",
        routing_decision: ApiRouteDecision | dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        route = ApiRouteDecision.model_validate(routing_decision)
        if route.workflow != "request_execution":
            raise ValueError("The model decision did not select request execution.")
        evidence = self._validate_route(route)
        if (
            evidence.integration_id != kwargs.get("integration_id")
            or evidence.operation_id != kwargs.get("operation_id")
        ):
            raise ValueError(
                "Request identifiers do not match the validated model operation decision."
            )
        request = self._build_validated(**kwargs).model_copy(
            update={
                "session_id": session_id,
                "routing_task_intent": route.task_intent,
            }
        )
        preview = self.builder.preview(request)
        parsed = urlsplit(request.url)
        hostname = (parsed.hostname or "").rstrip(".").lower()
        policy = self.executor.network_policy
        needs_host_approval = bool(
            policy.allowed_hosts
            and hostname
            not in {item.rstrip(".").lower() for item in policy.allowed_hosts}
        )
        needs_http_approval = parsed.scheme == "http" and not policy.allow_http
        if needs_host_approval or needs_http_approval:
            validation_policy = policy.model_copy(
                update={
                    "allowed_hosts": tuple(sorted({*policy.allowed_hosts, hostname})),
                    "allow_http": bool(policy.allow_http or needs_http_approval),
                }
            )
            validate_network_target(request.url, validation_policy)
            request = request.model_copy(
                update={
                    "approved_network_host": hostname,
                    "allow_insecure_http_once": needs_http_approval,
                }
            )
            reasons = []
            if needs_host_approval:
                reasons.append(f"host {hostname!r} is outside the configured API allowlist")
            if needs_http_approval:
                reasons.append("the request uses unencrypted HTTP")
            preview = preview.model_copy(
                update={
                    "approval_required": True,
                    "expected_side_effects": (
                        preview.expected_side_effects
                        + " Network-policy approval is required because "
                        + " and ".join(reasons)
                        + ". Approval applies once to this exact request."
                    ),
                }
            )
        publish_api_event(
            "api.request.validation.completed",
            {
                "integration_id": request.integration_id,
                "operation_id": request.operation_id,
                "valid": True,
                "routing_evidence": evidence.model_dump(mode="json"),
            },
        )
        result = self.executor.execute(
            request,
            preview=preview,
            approval_reference=approval_reference,
        )
        if not result.upstream_ok:
            if self.registry.get(request.integration_id).ephemeral:
                self.registry.discard_ephemeral(request.integration_id)
            raise UpstreamApiError(
                f"The upstream API returned HTTP {result.status_code}.",
                details={
                    "executed": True,
                    "integration_id": result.integration_id,
                    "operation_id": result.operation_id,
                    "status_code": result.status_code,
                    "content_type": result.content_type,
                    "body_kind": result.body_kind,
                    "json_body": result.json_body,
                    "text_body": result.text_body[:4000],
                    "latency_ms": result.latency_ms,
                },
            )
        if self.registry.get(request.integration_id).ephemeral:
            self.registry.discard_ephemeral(request.integration_id)
        self.discovery.record_success(
            task_intent=route.task_intent,
            integration_id=request.integration_id,
            operation_id=request.operation_id,
        )
        return result.model_dump(mode="json")

    def decide_approval(
        self,
        request_id: str,
        *,
        session_id: str,
        approve: bool,
        client_type: str,
    ) -> dict[str, Any]:
        if not approve:
            self.approvals.deny(
                request_id,
                session_id=session_id,
                client_type=client_type,
            )
            return {"approved": False, "executed": False}
        request, preview = self.approvals.approve(
            request_id,
            session_id=session_id,
            client_type=client_type,
        )
        result = self.executor.execute(
            request,
            preview=preview,
            approval_reference=request_id,
        )
        if self.registry.get(request.integration_id).ephemeral:
            self.registry.discard_ephemeral(request.integration_id)
        if result.upstream_ok:
            self.discovery.record_success(
                task_intent=request.routing_task_intent,
                integration_id=request.integration_id,
                operation_id=request.operation_id,
            )
        return {"approved": True, "executed": True, "result": result.model_dump(mode="json")}

    def _validate_route(
        self,
        routing_decision: ApiRouteDecision | dict[str, Any],
    ):
        decision = ApiRouteDecision.model_validate(routing_decision)
        candidates = self.discovery.search(decision.task_intent, limit=50)
        return self.discovery.validate_decision(decision, candidates=candidates)

    def _build_validated(self, **kwargs: Any):
        try:
            return self.builder.build(**kwargs)
        except ApiManagerError as exc:
            publish_api_event(
                "api.request.validation.failed",
                {
                    "integration_id": str(kwargs.get("integration_id") or ""),
                    "operation_id": str(kwargs.get("operation_id") or ""),
                    "error_code": exc.code,
                },
            )
            raise


def safe_result(operation: Any) -> dict[str, Any]:
    try:
        return {"ok": True, "result": operation()}
    except ApiManagerError as exc:
        return {"ok": False, **exc.to_dict()}
    except (ValueError, PermissionError) as exc:
        return {
            "ok": False,
            "error_code": "api_manager_validation_error",
            "message": str(exc),
        }


def _csv(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())
