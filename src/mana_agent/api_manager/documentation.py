"""Safe documentation ingestion and OpenAPI normalization."""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field

from mana_agent.api_manager.errors import (
    MalformedSpecificationError,
    UnresolvedSchemaReferenceError,
    UnsupportedDocumentationError,
)
from mana_agent.api_manager.models import (
    ApiIntegration,
    ApiOperation,
    ApiParameter,
    ApiServer,
    AuthenticationConfig,
    AuthenticationType,
    DocumentationSource,
    DocumentationSourceType,
    HttpMethod,
    OAuthFlow,
    OperationRiskLevel,
    ParameterLocation,
    RequestBody,
    ResponseDefinition,
    stable_id,
)


HTTP_METHODS = {"get", "head", "options", "post", "put", "patch", "delete"}


class DocumentationFetcher(Protocol):
    def fetch(self, url: str) -> tuple[bytes, str]: ...


class SemanticDefinition(BaseModel):
    """Model-produced semantic extraction; strict validation prevents invention."""

    model_config = ConfigDict(extra="forbid")
    servers: list[ApiServer] = Field(min_length=1)
    operations: list[ApiOperation] = Field(min_length=1)
    authentication: list[AuthenticationConfig] = Field(default_factory=list)
    description: str = ""


class DocumentationImporter:
    def __init__(
        self,
        *,
        allowed_file_roots: tuple[Path, ...] = (),
        fetcher: DocumentationFetcher | None = None,
    ) -> None:
        self.allowed_file_roots = tuple(path.expanduser().resolve() for path in allowed_file_roots)
        self.fetcher = fetcher

    def from_file(
        self,
        path: str | Path,
        *,
        name: str,
        source_decision_id: str,
        semantic_definition: SemanticDefinition | dict[str, Any] | None = None,
        evidence_text: str = "",
        evidence_documentation_ref: str = "",
    ) -> ApiIntegration:
        resolved = Path(path).expanduser().resolve()
        if self.allowed_file_roots and not any(
            resolved == root or root in resolved.parents for root in self.allowed_file_roots
        ):
            raise UnsupportedDocumentationError(
                "Documentation file is outside the authorized roots.",
                details={"path": str(resolved)},
            )
        if not resolved.is_file():
            raise UnsupportedDocumentationError("Documentation file was not found.")
        content = resolved.read_bytes()
        suffix = resolved.suffix.lower()
        hint = "json" if suffix == ".json" else "yaml" if suffix in {".yaml", ".yml"} else "text"
        return self.from_bytes(
            content,
            name=name,
            reference=str(resolved),
            source_type=DocumentationSourceType.LOCAL_FILE,
            format_hint=hint,
            source_decision_id=source_decision_id,
            semantic_definition=semantic_definition,
            evidence_text=evidence_text,
            evidence_documentation_ref=evidence_documentation_ref,
        )

    def from_url(
        self,
        url: str,
        *,
        name: str,
        source_decision_id: str,
        semantic_definition: SemanticDefinition | dict[str, Any] | None = None,
        evidence_text: str = "",
        evidence_documentation_ref: str = "",
    ) -> ApiIntegration:
        if self.fetcher is None:
            raise UnsupportedDocumentationError(
                "A policy-controlled documentation fetcher is required for URL imports."
            )
        content, content_type = self.fetcher.fetch(url)
        lowered = content_type.lower()
        hint = (
            "json"
            if "json" in lowered
            else "yaml"
            if "yaml" in lowered
            else "html"
            if "html" in lowered
            else "text"
        )
        return self.from_bytes(
            content,
            name=name,
            reference=url,
            source_type=DocumentationSourceType.WEBPAGE,
            format_hint=hint,
            source_decision_id=source_decision_id,
            semantic_definition=semantic_definition,
            evidence_text=evidence_text,
            evidence_documentation_ref=evidence_documentation_ref,
        )

    def from_text(
        self,
        content: str,
        *,
        name: str,
        source_decision_id: str,
        reference: str = "pasted-text",
        semantic_definition: SemanticDefinition | dict[str, Any] | None = None,
        evidence_text: str = "",
        evidence_documentation_ref: str = "",
    ) -> ApiIntegration:
        return self.from_bytes(
            content.encode("utf-8"),
            name=name,
            reference=reference,
            source_type=DocumentationSourceType.PASTED,
            format_hint="text",
            source_decision_id=source_decision_id,
            semantic_definition=semantic_definition,
            evidence_text=evidence_text or content,
            evidence_documentation_ref=evidence_documentation_ref,
        )

    def from_bytes(
        self,
        content: bytes,
        *,
        name: str,
        reference: str,
        source_type: DocumentationSourceType,
        format_hint: str,
        source_decision_id: str,
        semantic_definition: SemanticDefinition | dict[str, Any] | None = None,
        evidence_text: str = "",
        evidence_documentation_ref: str = "",
    ) -> ApiIntegration:
        if len(content) > 10 * 1024 * 1024:
            raise UnsupportedDocumentationError("Documentation exceeds the 10 MiB import limit.")
        digest = hashlib.sha256(content).hexdigest()
        text = content.decode("utf-8", errors="strict")
        formal = self._parse_formal(text, format_hint=format_hint)
        if formal is not None:
            formal_document, formal_format = formal
            integration, detected_type = normalize_openapi(
                formal_document,
                name=name,
                reference=reference,
                source_decision_id=source_decision_id,
                content_sha256=digest,
                source_format=formal_format,
                evidence_text=evidence_text or text,
                evidence_documentation_ref=evidence_documentation_ref,
            )
            source = integration.documentation_sources[0].model_copy(update={"type": detected_type})
            return integration.model_copy(update={"documentation_sources": (source,)})

        if semantic_definition is None:
            raise UnsupportedDocumentationError(
                "Unstructured API documentation requires a validated model semantic extraction. "
                "No heuristic or fallback extraction was executed."
            )
        normalized_text = _strip_html(text) if format_hint == "html" else text
        if not normalized_text.strip():
            raise UnsupportedDocumentationError("Documentation did not contain readable text.")
        semantic = SemanticDefinition.model_validate(semantic_definition)
        operations = _normalize_authentication_parameters(
            semantic.operations,
            integration_authentication=semantic.authentication,
        )
        _validate_inferred_operations(
            operations,
            documented_references=_documented_references(reference, normalized_text),
        )
        source = DocumentationSource(
            type=source_type,
            reference=reference,
            content_sha256=digest,
            source_decision_id=source_decision_id,
        )
        return ApiIntegration.create(
            name=name,
            description=semantic.description,
            servers=semantic.servers,
            operations=operations,
            authentication=semantic.authentication,
            documentation_source=source,
        )

    @staticmethod
    def _parse_formal(
        text: str,
        *,
        format_hint: str,
    ) -> tuple[dict[str, Any], str] | None:
        candidates = [format_hint]
        if format_hint == "text":
            candidates.extend(["json", "yaml"])
        for candidate in candidates:
            try:
                value = json.loads(text) if candidate == "json" else yaml.safe_load(text) if candidate == "yaml" else None
            except (json.JSONDecodeError, yaml.YAMLError):
                continue
            if isinstance(value, dict) and ("openapi" in value or "swagger" in value):
                return value, candidate
        return None


def _classify_reference(reference: str) -> tuple[str, str]:
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return "external", str(reference)
    path = reference[2:]
    tokens = [t.replace("~1", "/").replace("~0", "~") for t in path.split("/")]
    if len(tokens) >= 3 and tokens[0] == "components":
        if tokens[1] == "parameters":
            return "parameter", tokens[2]
        elif tokens[1] == "schemas":
            return "schema", tokens[2]
        elif tokens[1] == "requestBodies":
            return "requestBody", tokens[2]
        elif tokens[1] == "responses":
            return "response", tokens[2]
        elif tokens[1] == "securitySchemes":
            return "securityScheme", tokens[2]
        elif tokens[1] == "headers":
            return "header", tokens[2]
        return tokens[1], tokens[2]
    elif len(tokens) >= 2:
        if tokens[0] == "parameters":
            return "parameter", tokens[1]
        elif tokens[0] == "definitions":
            return "schema", tokens[1]
        elif tokens[0] == "responses":
            return "response", tokens[1]
        elif tokens[0] == "securityDefinitions":
            return "securityScheme", tokens[1]
        return tokens[0], tokens[1]
    return "other", tokens[-1] if tokens else ""


def _extract_documented_parameter(name: str, evidence_text: str) -> dict[str, Any] | None:
    if not evidence_text or not name:
        return None
    escaped_name = re.escape(name)
    pattern = re.compile(rf"\b{escaped_name}\b", re.IGNORECASE)
    if not pattern.search(evidence_text):
        return None

    lines = evidence_text.splitlines()
    target_block = []
    for idx, line in enumerate(lines):
        if pattern.search(line):
            start = max(0, idx - 2)
            end = min(len(lines), idx + 20)
            target_block = lines[start:end]
            break

    block_text = "\n".join(target_block) if target_block else evidence_text

    location = None
    loc_match = re.search(
        r"(?:location|in|param(?:eter)?\s*type|placed\s*in)\s*[:=\-]?\s*['\"]?(header|query|path|cookie)['\"]?",
        block_text,
        re.IGNORECASE,
    )
    if loc_match:
        location = loc_match.group(1).lower()
    elif re.search(r"\bheader\b", block_text, re.IGNORECASE):
        location = "header"
    elif re.search(r"\bquery\b", block_text, re.IGNORECASE):
        location = "query"
    elif re.search(r"\bpath\b", block_text, re.IGNORECASE):
        location = "path"
    elif re.search(r"\bcookie\b", block_text, re.IGNORECASE):
        location = "cookie"
    else:
        if name.lower() in {
            "accept-encoding",
            "accept",
            "authorization",
            "content-type",
            "user-agent",
            "x-api-key",
        }:
            location = "header"
        else:
            location = "query"

    required = False
    req_match = re.search(
        r"\brequired\s*[:=\-]?\s*(true|yes|1|required|mandatory)\b",
        block_text,
        re.IGNORECASE,
    )
    if req_match and not re.search(
        r"\b(?:optional|not\s*required|required\s*[:=\-]?\s*(?:false|no|0))\b",
        block_text,
        re.IGNORECASE,
    ):
        required = True

    param_type = "string"
    type_match = re.search(
        r"(?:type|data\s*type|schema)\s*[:=\-]?\s*['\"]?(string|integer|int|boolean|bool|number|float)['\"]?",
        block_text,
        re.IGNORECASE,
    )
    if type_match:
        raw_type = type_match.group(1).lower()
        param_type = (
            "integer"
            if raw_type in {"int", "integer"}
            else "boolean"
            if raw_type in {"bool", "boolean"}
            else "number"
            if raw_type in {"number", "float"}
            else "string"
        )

    description = ""
    desc_match = re.search(
        r"(?:description|desc|summary|about)\s*[:=\-]?\s*['\"]?([^\n\r]+)",
        block_text,
        re.IGNORECASE,
    )
    if desc_match:
        description = desc_match.group(1).strip().strip("'\"")
    else:
        for line in target_block:
            cleaned = line.strip()
            if (
                cleaned
                and not pattern.search(cleaned)
                and not any(
                    kw in cleaned.lower()
                    for kw in ("type:", "location:", "in:", "required:")
                )
            ):
                description = cleaned
                break

    schema: dict[str, Any] = {"type": param_type}
    enum_match = re.search(
        r"(?:supported\s*values?|values?|enum)\s*[:=\-]?\s*([^\n\r]+)",
        block_text,
        re.IGNORECASE,
    )
    if enum_match:
        raw_enum = enum_match.group(1).strip()
        enum_tokens = [
            t.strip().strip("'\"[]")
            for t in re.split(r"[,/|]|\bor\b|\band\b", raw_enum)
            if t.strip().strip("'\"[]")
        ]
        if enum_tokens and len(enum_tokens) <= 20:
            schema["enum"] = enum_tokens

    return {
        "name": name,
        "in": location,
        "required": required,
        "description": description or f"Documented parameter {name}",
        "schema": schema,
    }


def normalize_openapi(
    document: dict[str, Any],
    *,
    name: str,
    reference: str,
    source_decision_id: str,
    content_sha256: str,
    source_format: str = "",
    evidence_text: str = "",
    evidence_documentation_ref: str = "",
) -> tuple[ApiIntegration, DocumentationSourceType]:
    openapi_version = str(document.get("openapi") or "")
    swagger_version = str(document.get("swagger") or "")
    if openapi_version and not openapi_version.startswith("3."):
        raise UnsupportedDocumentationError(f"Unsupported OpenAPI version: {openapi_version}")
    if swagger_version and swagger_version != "2.0":
        raise UnsupportedDocumentationError(f"Unsupported Swagger version: {swagger_version}")
    if not openapi_version and not swagger_version:
        raise MalformedSpecificationError("The document is not OpenAPI or Swagger.")
    resolver = _LocalReferenceResolver(
        document,
        evidence_text=evidence_text,
        evidence_documentation_ref=evidence_documentation_ref,
        source_reference=reference,
    )
    is_swagger = bool(swagger_version)
    servers = _servers(document, is_swagger=is_swagger, reference=reference)
    security_schemes = _security_schemes(document, resolver=resolver, is_swagger=is_swagger)
    default_security = document.get("security")
    operations: list[ApiOperation] = []
    paths = document.get("paths")
    if not isinstance(paths, dict):
        raise MalformedSpecificationError("Specification paths must be an object.")
    for raw_path, path_item_value in paths.items():
        path_item = resolver.resolve_object(path_item_value, context=f"paths.{raw_path}")
        shared_parameters = path_item.get("parameters") or []
        for method_name, raw_operation in path_item.items():
            if method_name.lower() not in HTTP_METHODS:
                continue
            method = HttpMethod(method_name.upper())
            operation = resolver.resolve_object(raw_operation, context=f"{method.value} {raw_path}")
            all_raw_parameters = [*shared_parameters, *(operation.get("parameters") or [])]
            op_parameters = _parameters(all_raw_parameters, resolver=resolver)
            request_body = (
                _swagger_request_body(all_raw_parameters, resolver=resolver)
                if is_swagger
                else _request_body(operation.get("requestBody"), resolver=resolver)
            )
            auth, scopes = _operation_auth(
                operation.get("security", default_security),
                security_schemes,
            )
            operation_id = str(operation.get("operationId") or "").strip() or stable_id(
                "op", servers[0].url, method.value, str(raw_path)
            )
            source_pointer = f"{reference}#/paths/{_pointer_escape(str(raw_path))}/{method_name.lower()}"
            operations.append(
                ApiOperation(
                    operation_id=operation_id,
                    name=str(operation.get("summary") or operation_id),
                    description=str(operation.get("description") or ""),
                    method=method,
                    path=str(raw_path),
                    base_url=servers[0].url,
                    tags=tuple(str(item) for item in operation.get("tags") or ()),
                    parameters=tuple(op_parameters),
                    request_body=request_body,
                    responses=tuple(_responses(operation.get("responses") or {}, resolver=resolver, is_swagger=is_swagger)),
                    authentication=tuple(auth),
                    required_scopes=tuple(scopes),
                    risk_level=_risk_for(method),
                    source_reference=source_pointer,
                )
            )
    if not operations:
        raise MalformedSpecificationError("Specification does not define any supported operations.")
    description = str((document.get("info") or {}).get("description") or "")
    extension = source_format or ("json" if reference.lower().endswith(".json") else "yaml")
    source_type = (
        DocumentationSourceType.SWAGGER_JSON
        if is_swagger and extension == "json"
        else DocumentationSourceType.SWAGGER_YAML
        if is_swagger
        else DocumentationSourceType.OPENAPI_JSON
        if extension == "json"
        else DocumentationSourceType.OPENAPI_YAML
    )
    source = DocumentationSource(
        type=source_type,
        reference=reference,
        content_sha256=content_sha256,
        source_decision_id=source_decision_id,
    )
    metadata = (
        {"recovered_references": resolver.recovered_references}
        if resolver.recovered_references
        else {}
    )
    integration = ApiIntegration.create(
        name=name,
        description=description,
        servers=servers,
        operations=operations,
        authentication=list(security_schemes.values()),
        documentation_source=source,
        metadata=metadata,
    )
    return integration, source_type


class _LocalReferenceResolver:
    def __init__(
        self,
        document: dict[str, Any],
        *,
        evidence_text: str = "",
        evidence_documentation_ref: str = "",
        source_reference: str = "",
    ) -> None:
        self.document = document
        self.evidence_text = evidence_text
        self.evidence_documentation_ref = evidence_documentation_ref
        self.source_reference = source_reference
        self.recovered_references: list[dict[str, Any]] = []

    def resolve_object(self, value: Any, *, context: str) -> dict[str, Any]:
        resolved = self.resolve(value, seen=(), context=context)
        if not isinstance(resolved, dict):
            raise MalformedSpecificationError(f"{context} must resolve to an object.")
        return resolved

    def resolve(self, value: Any, *, seen: tuple[str, ...], context: str) -> Any:
        if isinstance(value, list):
            return [self.resolve(item, seen=seen, context=context) for item in value]
        if not isinstance(value, dict):
            return value
        reference = value.get("$ref")
        if reference:
            if not isinstance(reference, str) or not reference.startswith("#/"):
                raise UnresolvedSchemaReferenceError(
                    "Only local OpenAPI references are allowed.",
                    details={
                        "error_code": "openapi_local_ref_unresolved",
                        "reference": str(reference),
                        "reference_kind": "external",
                        "reference_name": "",
                        "context": context,
                        "source_reference": self.source_reference,
                        "recoverable": False,
                    },
                )
            if reference in seen:
                # Keep recursive schemas as local references rather than recursing forever.
                return {"$ref": reference}
            target: Any = self.document
            ref_kind, ref_name = _classify_reference(reference)
            try:
                for token in reference[2:].split("/"):
                    token = token.replace("~1", "/").replace("~0", "~")
                    target = target[token]
            except (KeyError, TypeError) as exc:
                if ref_kind == "parameter" and self.evidence_text:
                    recovered = _extract_documented_parameter(ref_name, self.evidence_text)
                    if recovered is not None:
                        self.recovered_references.append(
                            {
                                "reference": reference,
                                "evidence_documentation_ref": self.evidence_documentation_ref,
                                "recovery_type": "documented_parameter",
                            }
                        )
                        target = recovered
                    else:
                        raise UnresolvedSchemaReferenceError(
                            f"Local schema reference could not be resolved: {reference}",
                            details={
                                "error_code": "openapi_local_ref_unresolved",
                                "reference": reference,
                                "reference_kind": ref_kind,
                                "reference_name": ref_name,
                                "context": context,
                                "source_reference": self.source_reference,
                                "recoverable": False,
                            },
                        ) from exc
                else:
                    raise UnresolvedSchemaReferenceError(
                        f"Local schema reference could not be resolved: {reference}",
                        details={
                            "error_code": "openapi_local_ref_unresolved",
                            "reference": reference,
                            "reference_kind": ref_kind,
                            "reference_name": ref_name,
                            "context": context,
                            "source_reference": self.source_reference,
                            "recoverable": False,
                        },
                    ) from exc
            merged = dict(self.resolve(target, seen=(*seen, reference), context=context))
            merged.update({key: item for key, item in value.items() if key != "$ref"})
            value = merged
        return {
            key: self.resolve(item, seen=seen, context=f"{context}.{key}")
            for key, item in value.items()
        }


def _servers(
    document: dict[str, Any],
    *,
    is_swagger: bool,
    reference: str,
) -> list[ApiServer]:
    if is_swagger:
        schemes = document.get("schemes") or ["https"]
        host = str(document.get("host") or "").strip()
        if not host:
            raise MalformedSpecificationError("Swagger 2.0 requires a host.")
        base_path = str(document.get("basePath") or "").rstrip("/")
        return [ApiServer(url=f"{schemes[0]}://{host}{base_path}")]
    raw_servers = document.get("servers") or []
    if not raw_servers:
        raise MalformedSpecificationError(
            "OpenAPI specification requires an explicit server; no base URL was inferred."
        )
    result: list[ApiServer] = []
    for raw in raw_servers:
        if not isinstance(raw, dict) or not raw.get("url"):
            raise MalformedSpecificationError("OpenAPI server entries require a URL.")
        variables = {
            str(key): str((value or {}).get("default") or "")
            for key, value in (raw.get("variables") or {}).items()
        }
        missing_defaults = sorted(key for key, value in variables.items() if not value)
        if missing_defaults:
            raise MalformedSpecificationError(
                "OpenAPI server variables require explicit defaults: "
                + ", ".join(missing_defaults)
            )
        url = str(raw["url"])
        for key, value in variables.items():
            url = url.replace("{" + key + "}", value)
        if not urlsplit(url).scheme:
            if urlsplit(reference).scheme not in {"http", "https"}:
                raise MalformedSpecificationError(
                    "Relative OpenAPI server URLs require an HTTP(S) documentation source."
                )
            url = urljoin(reference, url)
        result.append(ApiServer(url=url, description=str(raw.get("description") or ""), variables=variables))
    return result


def _security_schemes(
    document: dict[str, Any],
    *,
    resolver: _LocalReferenceResolver,
    is_swagger: bool,
) -> dict[str, AuthenticationConfig]:
    raw_schemes = (
        document.get("securityDefinitions") or {}
        if is_swagger
        else (document.get("components") or {}).get("securitySchemes") or {}
    )
    schemes: dict[str, AuthenticationConfig] = {}
    for name, raw_value in raw_schemes.items():
        raw = resolver.resolve_object(raw_value, context=f"securitySchemes.{name}")
        kind = str(raw.get("type") or "").lower()
        location = str(raw.get("in") or "").lower()
        scheme = str(raw.get("scheme") or "").lower()
        if kind == "apikey":
            auth_type = AuthenticationType.API_KEY_QUERY if location == "query" else AuthenticationType.API_KEY_HEADER
        elif kind == "http" and scheme == "bearer":
            auth_type = AuthenticationType.BEARER
        elif kind == "http" and scheme == "basic" or is_swagger and kind == "basic":
            auth_type = AuthenticationType.BASIC
        elif kind in {"oauth2", "oauth"}:
            auth_type = AuthenticationType.OAUTH2
        else:
            continue
        flows: dict[str, OAuthFlow] = {}
        raw_flows = raw.get("flows") or {}
        if is_swagger and auth_type is AuthenticationType.OAUTH2:
            raw_flows = {
                str(raw.get("flow") or "authorizationCode"): {
                    "authorizationUrl": raw.get("authorizationUrl"),
                    "tokenUrl": raw.get("tokenUrl"),
                    "scopes": raw.get("scopes") or {},
                }
            }
        for flow_name, flow in raw_flows.items():
            flows[str(flow_name)] = OAuthFlow(
                authorization_url=str((flow or {}).get("authorizationUrl") or ""),
                token_url=str((flow or {}).get("tokenUrl") or ""),
                refresh_url=str((flow or {}).get("refreshUrl") or ""),
                scopes={str(key): str(value) for key, value in ((flow or {}).get("scopes") or {}).items()},
            )
        schemes[str(name)] = AuthenticationConfig(
            type=auth_type,
            scheme_name=str(name),
            parameter_name=str(raw.get("name") or ""),
            oauth_flows=flows,
            required=True,
        )
    return schemes


def _operation_auth(
    security: Any,
    schemes: dict[str, AuthenticationConfig],
) -> tuple[list[AuthenticationConfig], list[str]]:
    if security == []:
        return [AuthenticationConfig()], []
    if not security:
        return [], []
    selected: list[AuthenticationConfig] = []
    scopes: list[str] = []
    for requirement in security:
        if not isinstance(requirement, dict):
            continue
        for name, required_scopes in requirement.items():
            if name in schemes:
                selected.append(schemes[name])
                scopes.extend(str(scope) for scope in required_scopes or ())
    return selected, scopes


def _parameters(values: Any, *, resolver: _LocalReferenceResolver) -> list[ApiParameter]:
    result: list[ApiParameter] = []
    for index, raw_value in enumerate(values or []):
        raw = resolver.resolve_object(raw_value, context=f"parameters[{index}]")
        location = str(raw.get("in") or "")
        if location == "body":
            continue
        try:
            normalized_location = ParameterLocation(location)
        except ValueError as exc:
            raise MalformedSpecificationError(f"Unsupported parameter location: {location}") from exc
        schema = raw.get("schema") or {
            key: raw[key]
            for key in ("type", "format", "items", "enum", "default")
            if key in raw
        }
        result.append(
            ApiParameter(
                name=str(raw.get("name") or ""),
                location=normalized_location,
                required=bool(raw.get("required")),
                description=str(raw.get("description") or ""),
                schema=resolver.resolve(schema, seen=(), context=f"parameter.{raw.get('name')}"),
                style=str(raw.get("style") or ""),
                explode=raw.get("explode"),
            )
        )
    return result


def _swagger_request_body(
    parameters: list[Any],
    *,
    resolver: _LocalReferenceResolver,
) -> RequestBody | None:
    body: dict[str, Any] | None = None
    for index, value in enumerate(parameters):
        candidate = resolver.resolve_object(value, context=f"parameters[{index}]")
        if str(candidate.get("in") or "") == "body":
            body = candidate
            break
    if body is None:
        return None
    return RequestBody(
        required=bool(body.get("required")),
        description=str(body.get("description") or ""),
        content={
            "application/json": resolver.resolve(
                body.get("schema") or {},
                seen=(),
                context="body.schema",
            )
        },
    )


def _request_body(value: Any, *, resolver: _LocalReferenceResolver) -> RequestBody | None:
    if value is None:
        return None
    raw = resolver.resolve_object(value, context="requestBody")
    content = {
        str(media_type): resolver.resolve((definition or {}).get("schema") or {}, seen=(), context=f"requestBody.{media_type}")
        for media_type, definition in (raw.get("content") or {}).items()
    }
    return RequestBody(
        required=bool(raw.get("required")),
        description=str(raw.get("description") or ""),
        content=content,
    )


def _responses(values: Any, *, resolver: _LocalReferenceResolver, is_swagger: bool) -> list[ResponseDefinition]:
    result: list[ResponseDefinition] = []
    for status, raw_value in values.items():
        raw = resolver.resolve_object(raw_value, context=f"responses.{status}")
        content = raw.get("content") or {}
        if is_swagger and raw.get("schema"):
            content = {"application/json": {"schema": raw.get("schema")}}
        schemas = {
            str(media_type): resolver.resolve((definition or {}).get("schema") or {}, seen=(), context=f"response.{status}")
            for media_type, definition in content.items()
        }
        result.append(
            ResponseDefinition(
                status_code=str(status),
                description=str(raw.get("description") or ""),
                content=schemas,
                headers={
                    str(key): resolver.resolve(value, seen=(), context=f"response.{status}.headers")
                    for key, value in (raw.get("headers") or {}).items()
                },
            )
        )
    return result


def _risk_for(method: HttpMethod) -> OperationRiskLevel:
    if method in {HttpMethod.GET, HttpMethod.HEAD, HttpMethod.OPTIONS}:
        return OperationRiskLevel.READ_ONLY
    if method is HttpMethod.POST:
        return OperationRiskLevel.CREATE
    if method in {HttpMethod.PUT, HttpMethod.PATCH}:
        return OperationRiskLevel.UPDATE
    if method is HttpMethod.DELETE:
        return OperationRiskLevel.DELETE
    return OperationRiskLevel.HIGH


def _documented_references(reference: str, text: str) -> set[str]:
    references = {reference}
    for match in re.finditer(r"https?://[^\s<>\"']+", text):
        documented_url = match.group(0).rstrip(".,;:!?)]}")
        if documented_url:
            references.add(documented_url)
    return references


def _normalize_authentication_parameters(
    operations: list[ApiOperation],
    *,
    integration_authentication: list[AuthenticationConfig],
) -> list[ApiOperation]:
    normalized: list[ApiOperation] = []
    for operation in operations:
        authentication = [*integration_authentication, *operation.authentication]
        authentication_parameters: set[tuple[ParameterLocation, str]] = set()
        for auth in authentication:
            if auth.type is AuthenticationType.API_KEY_QUERY:
                authentication_parameters.add((ParameterLocation.QUERY, auth.parameter_name))
            elif auth.type is AuthenticationType.API_KEY_HEADER:
                authentication_parameters.add(
                    (ParameterLocation.HEADER, auth.parameter_name.lower())
                )
            elif auth.type is AuthenticationType.CUSTOM_HEADERS:
                authentication_parameters.update(
                    (ParameterLocation.HEADER, name.lower()) for name in auth.custom_header_names
                )
        parameters = tuple(
            parameter
            for parameter in operation.parameters
            if (
                parameter.location,
                parameter.name.lower()
                if parameter.location is ParameterLocation.HEADER
                else parameter.name,
            )
            not in authentication_parameters
        )
        normalized.append(operation.model_copy(update={"parameters": parameters}, deep=True))
    return normalized


def _validate_inferred_operations(
    operations: list[ApiOperation],
    *,
    documented_references: set[str],
) -> None:
    for operation in operations:
        if not operation.source_reference:
            raise MalformedSpecificationError(
                "Every inferred operation requires a source reference."
            )
        if not operation.source_reference.startswith("#") and not any(
            operation.source_reference == reference
            or operation.source_reference.startswith(f"{reference}#")
            for reference in documented_references
        ):
            raise MalformedSpecificationError(
                "Inferred operation source references must cite the imported documentation."
            )
        for auth in operation.authentication:
            if auth.inferred and not auth.unresolved:
                raise MalformedSpecificationError(
                    "Inferred authentication must remain unresolved until explicitly configured."
                )


def _strip_html(text: str) -> str:
    without_scripts = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    return html.unescape(re.sub(r"(?s)<[^>]+>", " ", without_scripts))


def _pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")
